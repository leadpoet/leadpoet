from pathlib import Path

import pytest

from gateway.tee import protected_workflows as protected_workflows_module
from gateway.tee.protected_workflows import (
    PROTECTED_SYMBOLS,
    ProtectedWorkflowError,
    build_manifest,
    load_manifest,
    stage_external_protected_sources,
    verify_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "gateway/tee/protected_workflows.json"


def test_committed_protected_manifest_matches_current_source():
    verify_manifest(ROOT, load_manifest(MANIFEST_PATH))


def test_retired_trust_surfaces_are_absent_from_inventory():
    retired = {
        "gateway/tee/disable_gateway_miner_submissions_secret.py",
        "gateway/tee/gateway_miner_maintenance_restart_v1.py",
        "gateway/tee/inter_enclave_tls.py",
        "gateway/tee/provider_broker_v2.py",
        "gateway/utils/tee_egress_forwarder.py",
    }
    assert retired.isdisjoint(PROTECTED_SYMBOLS)


def test_current_release_and_runtime_identity_boundaries_remain_protected():
    assert {
        "materialize_gateway_code_hash_runtime",
        "iter_gateway_code_hash_payloads",
        "compute_gateway_code_hash",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/code_hash.py"])
    assert {
        "RuntimeIdentityV2",
        "_validate_public_configuration",
        "_validate_release_configuration",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/runtime_identity_v2.py"])
    assert {
        "validate_release_manifest",
        "validate_prior_release_manifest",
        "historical_two_role_specs",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/release_manifest_v2.py"])
    assert {
        "active_enclave_role",
        "allowed_exact_methods",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/rpc_authority.py"])
    assert {
        "handle_v2_runtime_rpc",
        "VSOCKRPCCleanupError",
        "vsock_rpc_transport_health",
        "_serve_vsock_connections",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/tee_service.py"])


def test_chain_and_normal_weight_signing_contracts_remain_protected():
    assert {
        "build_transport_attempt",
        "validate_transport_attempt",
        "build_boot_identity_body",
        "verify_boot_identity_nitro",
    } <= set(PROTECTED_SYMBOLS["leadpoet_canonical/attested_v2.py"])
    assert {
        "decode_weights_storage",
        "parse_finalized_header",
        "ss58_encode_account_id",
    } <= set(PROTECTED_SYMBOLS["leadpoet_canonical/chain_source_v2.py"])


def test_targeted_verifier_and_its_caller_remain_protected():
    assert {
        "_llm_reverify_company",
        "_run_targeted_company_evidence_investigation",
    } <= set(PROTECTED_SYMBOLS["qualification/scoring/lead_scorer.py"])


def test_enclave_surface_stages_every_external_protected_source(tmp_path: Path):
    enclave_root = tmp_path / "gateway"
    for relative_path in PROTECTED_SYMBOLS:
        if not relative_path.startswith("gateway/"):
            continue
        source = ROOT / relative_path
        destination = enclave_root / relative_path.split("/", 1)[1]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    staged_root = enclave_root / "_attested_runtime"
    expected = sum(not path.startswith("gateway/") for path in PROTECTED_SYMBOLS)
    assert stage_external_protected_sources(ROOT, staged_root) == expected
    verify_manifest(enclave_root, load_manifest(MANIFEST_PATH))


def test_protected_manifest_detects_policy_constant_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    policy_path = tmp_path / "policy.py"
    policy_path.write_text('POLICY_VERSION = "v1"\n', encoding="utf-8")
    monkeypatch.setattr(
        protected_workflows_module,
        "PROTECTED_SYMBOLS",
        {"policy.py": ("POLICY_VERSION",)},
    )
    manifest = build_manifest(
        tmp_path,
        baseline_commit="1" * 40,
        protected_source_commit="2" * 40,
    )
    policy_path.write_text('POLICY_VERSION = "v2"\n', encoding="utf-8")
    with pytest.raises(ProtectedWorkflowError, match="policy.py:POLICY_VERSION"):
        verify_manifest(tmp_path, manifest)
