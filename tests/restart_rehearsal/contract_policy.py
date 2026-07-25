"""Allowlist for strict adapters used by the isolated launcher rehearsal."""

from __future__ import annotations

from typing import Any


EXACT_CONTRACT_ADAPTER_MODULES = {
    "Leadpoet.utils.restart_epoch_gate",
    "gateway.tee.prepare_gateway_envelopes_v2",
    "gateway.tee.release_archive_v2",
    "gateway.tee.release_channel_v2",
    "gateway.tee.restart_preflight_v2",
    "gateway.tee.verify_v2_runtime_ready",
    "gateway.research_lab.provider_profiles_v2",
    "gateway.utils.tee_kms_provision_v2",
    "gateway.utils.tee_v2_bootstrap",
    "validator_tee.host.docker_operation_guard_v2",
    "validator_tee.host.hotkey_bootstrap_v2",
    "validator_tee.host.refresh_hotkey_config_v2",
    "validator_tee.host.release_archive_v2",
    "validator_tee.host.restart_preflight_v2",
    "validator_tee.host.runtime_v2_bootstrap",
    "validator_tee.host.verify_chain_signing_profile_v2",
    "validator_tee.host.verify_release_gate_v2",
    "validator_tee.scripts.stage_runtime_artifacts_v2",
}
EXACT_CONTRACT_ADAPTER_SCRIPTS = {
    "docker_image_normalizer_v2.py",
    "host_memory_guard_v2.py",
    "release_manifest_v2.py",
    "sandbox_runtime_artifact.py",
    "scoring_wheelhouse.py",
    "stage_runtime_artifacts_v2.py",
    "verify_release_artifacts_v2.py",
    "verify_topology.py",
}
EXACT_CONTRACT_ADAPTER_PROCESSES = {
    "gateway.main",
    "gateway.tee_egress",
    "gateway.tee_relay",
    "validator.chain_relay",
}
EXACT_CONTRACT_ADAPTER_BOUNDARIES = {
    "bash.build_drand_cabi_v2",
    "http.local_gateway",
    "host.containerd_state",
    "host.cpu_capacity",
    "host.filesystem_capacity",
    "host.memory_capacity",
    "host.mount_namespace",
    "host.process_lookup",
    "host.process_termination",
    "host.socket_state",
    "host.systemd",
    "host.timing",
    "python.scrub_parent_environment",
    "python.validator_coordinator",
    "python_dependencies.download",
    "python_dependencies.bootstrap",
    "python_dependencies.install",
    "python_dependencies.uninstall",
}
EXACT_EXTERNAL_ADAPTER_KINDS = {
    "aws",
    "curl",
    "docker",
    "nitro",
}
EXACT_WEIGHT_READINESS_BOUNDARIES = {
    "chain_epoch",
    "champion_reward_backfill",
    "cutover_readiness",
    "direct_allocation",
    "localhost_allocation_http",
    "settlement_backfill",
    "source_reward_backfill",
}
EXACT_CAPACITY_SUBSTITUTIONS = {
    "host.cpu_capacity",
    "host.memory_capacity",
}


def substitution_identity(row: dict[str, Any]) -> str:
    return str(
        row.get("substitution")
        or row.get("module")
        or row.get("script")
        or row.get("process")
        or ""
    )


def is_classified_contract_adapter(identity: str) -> bool:
    return (
        identity in EXACT_CONTRACT_ADAPTER_MODULES
        or identity in EXACT_CONTRACT_ADAPTER_SCRIPTS
        or identity in EXACT_CONTRACT_ADAPTER_PROCESSES
        or identity in EXACT_CONTRACT_ADAPTER_BOUNDARIES
    )


def is_classified_contract_fixture(row: dict[str, Any]) -> bool:
    kind = row.get("kind")
    if kind in EXACT_EXTERNAL_ADAPTER_KINDS:
        return isinstance(row.get("argv"), list)
    if kind == "weight-readiness-boundary":
        return row.get("boundary") in EXACT_WEIGHT_READINESS_BOUNDARIES
    if kind == "weight-readiness-persistence":
        attempts = row.get("attempts")
        return isinstance(attempts, list) and all(
            isinstance(attempt, dict)
            and attempt.get("method") in {"GET", "HEAD"}
            and isinstance(attempt.get("attempt_number"), int)
            for attempt in attempts
        )
    return False
