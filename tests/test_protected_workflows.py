import ast
import io
from pathlib import Path
import re
import subprocess
import tarfile

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
MANIFEST_PATH = ROOT / "gateway" / "tee" / "protected_workflows.json"
MINER_MAINTENANCE_TRUST_FILES = (
    "gateway/tee/disable_gateway_miner_submissions_secret.py",
    "gateway/tee/gateway_miner_maintenance_restart_v1.py",
    "gateway/tee/restart_preflight_v2.py",
    "scripts/verify_installed_gateway_controller_v1.py",
)


def _top_level_source_symbols(relative_path: str) -> set[str]:
    tree = ast.parse(
        (ROOT / relative_path).read_text(encoding="utf-8"),
        filename=relative_path,
    )
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            symbols.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
    return symbols | {"__module__"}


def test_miner_maintenance_protected_inventory_is_complete():
    for relative_path in MINER_MAINTENANCE_TRUST_FILES:
        assert set(PROTECTED_SYMBOLS[relative_path]) == _top_level_source_symbols(
            relative_path
        )


def test_committed_protected_workflow_manifest_matches_source(tmp_path: Path):
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
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", protected_source, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    archived = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            protected_source,
            "--",
            *sorted(PROTECTED_SYMBOLS),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        timeout=30,
    ).stdout
    protected_root = tmp_path / "protected-source"
    with tarfile.open(fileobj=io.BytesIO(archived), mode="r:") as archive:
        for relative_path in PROTECTED_SYMBOLS:
            member = archive.getmember(relative_path)
            assert member.isfile()
            source = archive.extractfile(member)
            assert source is not None
            destination = protected_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read())
    reproduced = build_manifest(
        protected_root,
        baseline_commit=manifest["baseline_commit"],
        protected_source_commit=protected_source,
    )
    assert reproduced == manifest
    assert len(manifest["entries"]) == sum(len(items) for items in PROTECTED_SYMBOLS.values())


def test_shared_docker_host_veto_and_source_add_lifecycle_are_protected():
    assert {
        "ATTESTED_RUNTIME_DIR",
        "ATTESTED_RUNTIME_PACKAGES",
        "ATTESTED_RUNTIME_FILES",
        "ATTESTED_RUNTIME_GENERATED_FILES",
        "_ATTESTED_RUNTIME_ROLES",
        "_FULL_COMMIT_RE",
        "_FALLBACK_COMMAND_TIMEOUT_SECONDS",
        "ROOT_FILES",
        "INCLUDE_DIRS",
        "HASH_SUFFIXES",
        "EXCLUDED_DIRS",
        "EXCLUDED_SUFFIXES",
        "EXCLUDED_NAMES",
        "GatewayCodeHashError",
        "_is_hashable",
        "_iter_files",
        "_fallback_environment",
        "_run_fallback_command",
        "_fallback_commit",
        "materialize_gateway_code_hash_runtime",
        "iter_gateway_code_hash_files",
        "iter_gateway_code_hash_payloads",
        "compute_gateway_code_hash",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/code_hash.py"])
    assert {
        "stage_external_protected_sources",
        "main",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/protected_workflows.py"])
    assert {
        "prepare_gateway_miner_maintenance_restart",
        "verify_gateway_miner_maintenance_state",
        "bootstrap_gateway_miner_maintenance_restart",
        "verify_gateway_miner_maintenance_runtime_state",
        "main",
    } <= set(
        PROTECTED_SYMBOLS[
            "gateway/tee/gateway_miner_maintenance_restart_v1.py"
        ]
    )
    assert {
        "verify_installed_controller_bundle",
        "_exec_verified_helper",
        "main",
    } <= set(
        PROTECTED_SYMBOLS[
            "scripts/verify_installed_gateway_controller_v1.py"
        ]
    )
    assert "main" in PROTECTED_SYMBOLS["gateway/tee/restart_preflight_v2.py"]
    assert {
        "_EXACT_HOST_GATEWAY_ARGS",
        "_HOST_GATEWAY_PYTHON_COMMAND",
        "_MAX_HOST_GATEWAY_CMDLINE_BYTES",
        "inspect_exact_host_gateway_runtime",
    } <= set(
        PROTECTED_SYMBOLS[
            "validator_tee/host/docker_operation_guard_v2.py"
        ]
    )
    assert {
        "docker_operation_admission_lock_path",
        "docker_operation_lock_path",
        "_acquire_file_lock_until",
        "wait_for_docker_daemon_ready",
        "shared_docker_operation_lock",
        "shared_docker_operation_source_paths",
    } <= set(PROTECTED_SYMBOLS["research_lab/docker_operation_lock_v2.py"])
    assert {
        "_docker_operation_admission_lock_file",
        "_docker_operation_lock_scope",
        "_run_sync_build_step_to_completion",
        "_communicate_build_process_to_completion",
    } <= set(PROTECTED_SYMBOLS["gateway/utils/pcr0_builder.py"])
    assert "build_source_add_sandbox_runner" in PROTECTED_SYMBOLS[
        "gateway/research_lab/source_add_trial_runner.py"
    ]


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
    expected_external_count = sum(
        not relative_path.startswith("gateway/")
        for relative_path in PROTECTED_SYMBOLS
    )
    assert (
        stage_external_protected_sources(ROOT, staged_root)
        == expected_external_count
    )
    assert (
        staged_root
        / "validator_tee"
        / "host"
        / "docker_operation_guard_v2.py"
    ).is_file()
    verify_manifest(enclave_root, load_manifest(MANIFEST_PATH))


def test_external_protected_source_staging_rejects_mismatched_existing_file(
    tmp_path: Path,
):
    staged = tmp_path / "staged"
    target = staged / "validator_tee" / "host" / "docker_operation_guard_v2.py"
    target.parent.mkdir(parents=True)
    target.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ProtectedWorkflowError, match="staged protected source differs"):
        stage_external_protected_sources(ROOT, staged)


def test_scoring_receipt_failure_policy_is_protected():
    assert {
        "_DIRECT_SUPABASE_SIDECAR_NAMESPACES",
        "_job_input_limit_bytes",
        "ExecutionContextV2.record_transport",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/execution_job_manager_v2.py"])
    assert "_local_failed_receipt_hashes" in PROTECTED_SYMBOLS[
        "gateway/research_lab/attested_scoring_v2.py"
    ]
def test_ancestry_unknown_commit_recovery_is_protected():
    assert {
        "_ANCESTRY_CHECKPOINT_UNKNOWN_COMMIT_BACKOFF_SECONDS",
        "_ancestry_checkpoint_unknown_commit_sleep",
        "_rehydrate_compact_execution_graph_v2",
        "load_execution_result_v2",
        "persist_ancestry_checkpoint_v2",
    } <= set(
        PROTECTED_SYMBOLS["gateway/research_lab/attested_v2_store.py"]
    )


def test_inter_enclave_replay_and_identity_boundaries_are_protected():
    assert {
        "_select_committed_encrypted_artifacts",
        "persist_execution_transport_artifacts_v2",
    } <= set(PROTECTED_SYMBOLS["gateway/research_lab/attested_artifacts_v2.py"])
    assert {
        "ATTESTED_TLS_CERTIFICATE_LIFETIME",
        "ATTESTED_TLS_CERTIFICATE_CLOCK_SKEW",
        "_atomic_private_write",
        "generate_ephemeral_tls_identity",
        "write_identity_to_tmpfs",
        "create_mutual_tls_context",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/mtls_identity.py"])
    assert {
        "REPLAY_WAIT_SECONDS",
        "TRANSPORT_HEALTH_SCHEMA_VERSION",
        "InterEnclaveTransportCleanupError",
        "_close_transport_required",
        "_RetryableInterEnclaveTransportError",
        "_recv_exact",
        "_send_frame",
        "_read_frame",
        "AttestedPeerRegistry",
        "AttestedTLSRPCClient",
        "AttestedTLSRPCServer",
        "build_rpc_request",
        "validate_rpc_request",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/inter_enclave_tls.py"])
    assert {
        "handle_v2_runtime_rpc",
        "get_v2_provider_broker",
        "get_v2_inter_enclave_client",
        "execute_v2_provider_request",
        "handle_inter_enclave_rpc",
        "start_v2_tls_service",
        "VSOCKRPCCleanupError",
        "_close_vsock_rpc_required",
        "_recover_vsock_rpc_cleanup_failures",
        "vsock_rpc_transport_health",
        "_handle_vsock_connection",
        "_serve_vsock_connections",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/tee_service.py"])
    assert {"TopologyHealthError", "verify_roles"} <= set(
        PROTECTED_SYMBOLS["gateway/tee/verify_topology.py"]
    )
    assert "_V2_RUNTIME_CONFIG_SCHEMA" in PROTECTED_SYMBOLS[
        "gateway/tee/verify_topology.py"
    ]


def test_artifact_egress_transport_boundaries_are_protected():
    assert {
        "_OBSERVED_EMPLOYEE_COUNT_INTERVALS",
        "normalize_observed_employee_count_bucket",
    } <= set(PROTECTED_SYMBOLS["research_lab/employee_buckets.py"])
    assert {
        "_decision_from_observed_employee_size",
        "_reverify_decision",
        "_llm_reverify_company",
    } <= set(PROTECTED_SYMBOLS["qualification/scoring/lead_scorer.py"])
    assert {
        "PROVIDER_TRANSPORT_HEALTH_SCHEMA_VERSION",
        "PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_SCHEMA_VERSION",
        "EGRESS_POLICY_DIRECT_ONLY",
        "_PROVIDER_TERMINAL_STATUSES",
        "_CHAIN_WEIGHT_OBSERVATION_PURPOSE",
        "_SAFE_ERROR_TYPE_RE",
        "_PROVIDER_ID_RE",
        "_PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_FIELDS",
        "_PROVIDER_TRANSPORT_FAILURE_STAGES",
        "_CLEANUP_RESOURCE_KIND_BY_STAGE",
        "_MAX_DIAGNOSTIC_ERRNO",
        "_EXPLICIT_HTTP_TRANSPORT_ATTRIBUTE",
        "_BROKER_OWNED_HTTPX_CLIENTS_LOCK",
        "_BROKER_OWNED_HTTPX_CLIENTS",
        "_BROKER_OWNED_HTTPX_SEND_GRANT",
        "_register_broker_owned_httpx_client",
        "is_broker_owned_httpx_client",
        "_broker_owned_httpx_send_scope",
        "_local_resource_failure",
        "_safe_error_type",
        "_failure_code",
        "validate_provider_transport_failure_diagnostic",
        "_provider_transport_failure_diagnostic",
        "_force_close_response_network_stream",
        "_close_client_transports",
        "ProviderTransportCleanupError",
        "ProviderRouteV2",
        "HTTPXProviderTransport",
        "ProviderBrokerV2.reseal_transport_failure_diagnostic",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/provider_broker_v2.py"])
    assert "BrokeredProviderTransportV2.install" in PROTECTED_SYMBOLS[
        "gateway/tee/provider_client_v2.py"
    ]
    assert {
        "lifespan",
        "_start_source_add_dispatcher_task",
        "_SOURCE_ADD_INDEPENDENT_PATHS",
        "_gateway_source_add_dispatcher_ready",
    } <= set(PROTECTED_SYMBOLS["gateway/main.py"])
    assert {
        "_TRANSIENT_ERROR_SIGNATURES",
        "_TRANSIENT_ERROR_TYPE_SIGNATURES",
        "_is_transient_store_error",
    } <= set(PROTECTED_SYMBOLS["gateway/research_lab/store.py"])
    assert {
        "COORDINATOR_ROLE",
        "active_enclave_role",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/rpc_authority.py"])
    assert {
        "_FAIL_CLOSED_REQUEST_SCHEMA_VERSION",
        "_SEMANTICS_HEALTH_STAGES",
        "ProviderSemanticsAuthorityV2",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/provider_semantics_v2.py"])
    # Host-only physical validation is not imported into the gateway enclave.
    # Its complete Git blob is bound by the production parity contract instead
    # of being represented as measured enclave AST symbols.
    assert "scripts/run_physical_v2_staging.py" not in PROTECTED_SYMBOLS
    assert {
        "TUNNEL_FRAMING_HEADER",
        "TUNNEL_FRAMING_MODE",
        "send_tunnel_frame",
        "receive_tunnel_frame",
        "relay_raw_and_framed",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/egress_framing.py"])
    assert {
        "policy_document",
        "destination_policy_hash",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/egress_policy.py"])
    assert {
        "MAX_PROC_TCP_HEALTH_BYTES",
        "MAX_PROC_TCP_HEALTH_ROWS",
        "_parse_proxy_request",
        "_proc_tcp_address_is_loopback",
        "_loopback_tcp_state_counts",
        "_process_transport_resource_health",
        "_shutdown_and_close_socket",
        "EnclaveEgressProxyCleanupError",
        "_FramedParentBridge",
        "_ManagedProxyStream",
        "EnclaveEgressProxy",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/egress_proxy.py"])
    assert {
        "ArtifactTransportCleanupError",
        "_ArtifactVerificationTransportPool",
        "_ArtifactVerificationTransportSession",
        "ArtifactPersistenceVerifierV2",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/artifact_persistence_v2.py"])
    assert {
        "TEEEgressForwarderCleanupError",
        "_handle_connection",
        "_connect_public_destination",
        "_shutdown_and_close_socket",
        "TEEEgressForwarder",
        "main",
    } <= set(PROTECTED_SYMBOLS["gateway/utils/tee_egress_forwarder.py"])
    assert {
        "vsock_rpc_transport_health_lock",
        "vsock_rpc_pending_cleanup_failures",
        "vsock_rpc_terminal_failure_event",
        "vsock_rpc_cleanup_recovery_lock",
        "_recover_vsock_rpc_cleanup_failures",
        "_serve_vsock_connections",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/tee_service.py"])
    assert {
        "_RETIRED_CLEANUP_LOCK",
        "_RETIRED_CLEANUP_RECOVERY_LOCK",
        "_RETIRED_CLEANUP_RESOURCES",
        "_retry_retired_cleanup",
    } <= set(
        PROTECTED_SYMBOLS["gateway/tee/proxy_transport_preflight_v2.py"]
    )


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
    target = copied_root / "research_lab" / "employee_buckets.py"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            "if isinstance(value, bool):",
            "if value is None:\n        return str(default or \"\")\n    if isinstance(value, bool):",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProtectedWorkflowError, match="employee_buckets.py"):
        verify_manifest(copied_root, manifest)


def test_protected_manifest_detects_policy_constant_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    copied_root = tmp_path / "repo"
    policy_path = copied_root / "policy.py"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text('POLICY_VERSION = "v1"\n', encoding="utf-8")
    monkeypatch.setattr(
        protected_workflows_module,
        "PROTECTED_SYMBOLS",
        {"policy.py": ("POLICY_VERSION",)},
    )
    manifest = build_manifest(
        copied_root,
        baseline_commit="1" * 40,
        protected_source_commit="2" * 40,
    )

    policy_path.write_text('POLICY_VERSION = "v2"\n', encoding="utf-8")

    with pytest.raises(ProtectedWorkflowError, match="policy.py:POLICY_VERSION"):
        verify_manifest(copied_root, manifest)
