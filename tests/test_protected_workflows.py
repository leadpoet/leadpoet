import ast
from pathlib import Path
import shutil
import subprocess
import sys

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


def test_protected_hashes_are_cross_version_and_fail_closed():
    python312 = shutil.which("python3.12")
    if python312 is None:
        pytest.skip("Python 3.12 is needed for the enclave hash compatibility check")
    script = """
import ast
import sys
from gateway.tee.protected_workflows import _symbol_hash

def digest(source):
    return _symbol_hash(ast.parse(source).body[0])

original = digest('def protected(value): return value')
changed = digest('def protected(value): return value + 1')
assert original != changed
literal_with_field_text = digest("def protected(): return 'a, type_params=[]b'")
literal_without_field_text = digest("def protected(): return 'ab'")
assert literal_with_field_text != literal_without_field_text
if sys.version_info >= (3, 12):
    generic_t = digest('def protected[T](value): return value')
    generic_u = digest('def protected[U](value): return value')
    assert generic_t != generic_u
print(original, changed)
"""
    host_hashes = subprocess.check_output(
        [sys.executable, "-c", script], cwd=ROOT, text=True
    ).strip()
    enclave_hashes = subprocess.check_output(
        [python312, "-c", script], cwd=ROOT, text=True
    ).strip()
    assert host_hashes == enclave_hashes


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
    assert "__module__" in PROTECTED_SYMBOLS["gateway/chain_startup.py"]
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
    assert {"validate_release_manifest"} <= set(
        PROTECTED_SYMBOLS["gateway/tee/release_manifest_v2.py"]
    )
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
        "_is_bound_homepage_name_conflict",
        "_run_targeted_company_evidence_investigation",
        "_settle_completed_company_investigation",
        "_verify_company_fit",
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


def test_extracted_investigation_settlement_remains_ast_protected(tmp_path, monkeypatch):
    relative = "qualification/scoring/lead_scorer.py"
    source = (ROOT / relative).read_text()
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(source)
    monkeypatch.setattr(protected_workflows_module, "PROTECTED_SYMBOLS", {
        relative: ("_settle_completed_company_investigation",),
    })
    manifest = build_manifest(
        tmp_path, baseline_commit="1" * 40, protected_source_commit="2" * 40,
    )
    verify_manifest(tmp_path, manifest)
    guard = 'and receipt.get("completed_submitted_findings") is True'
    settlement = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_settle_completed_company_investigation"
    )
    segment = ast.get_source_segment(source, settlement)
    assert segment.count(guard) == 1
    target.write_text(source.replace(segment, segment.replace(guard, 'and True', 1), 1))
    with pytest.raises(ProtectedWorkflowError, match="_settle_completed_company_investigation"):
        verify_manifest(tmp_path, manifest)


def test_final_company_fit_wrapper_remains_ast_protected(tmp_path, monkeypatch):
    relative = "qualification/scoring/lead_scorer.py"
    source = (ROOT / relative).read_text()
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(source)
    monkeypatch.setattr(protected_workflows_module, "PROTECTED_SYMBOLS", {
        relative: ("_verify_company_fit",),
    })
    manifest = build_manifest(
        tmp_path, baseline_commit="1" * 40, protected_source_commit="2" * 40,
    )
    verify_manifest(tmp_path, manifest)
    guard = 'and web_details.get("_completed_investigation_settled") is not True'
    assert source.count(guard) == 1
    target.write_text(source.replace(guard, 'and True', 1))
    with pytest.raises(ProtectedWorkflowError, match="_verify_company_fit"):
        verify_manifest(tmp_path, manifest)


def test_bound_homepage_identity_conflict_remains_ast_protected(tmp_path, monkeypatch):
    relative = "qualification/scoring/lead_scorer.py"
    source = (ROOT / relative).read_text()
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(source)
    monkeypatch.setattr(protected_workflows_module, "PROTECTED_SYMBOLS", {
        relative: ("_is_bound_homepage_name_conflict",),
    })
    manifest = build_manifest(
        tmp_path, baseline_commit="1" * 40, protected_source_commit="2" * 40,
    )
    verify_manifest(tmp_path, manifest)
    guard = 'and value["submitted_domain"] == value["observed_domain"]'
    assert source.count(guard) == 1
    target.write_text(source.replace(guard, "and True", 1))
    with pytest.raises(ProtectedWorkflowError, match="_is_bound_homepage_name_conflict"):
        verify_manifest(tmp_path, manifest)
