"""Fail-closed RPC authority for each physical gateway enclave role."""

from __future__ import annotations

import os
from typing import FrozenSet


COORDINATOR_ROLE = "gateway_coordinator"
SCORING_ROLES = frozenset({"gateway_scoring_a", "gateway_scoring_b"})
AUTORESEARCH_ROLE = "gateway_autoresearch"

COMMON_METHODS = frozenset(
    {
        "get_public_key",
        "get_attestation",
        "role_health",
        "v2_configure_runtime",
        "v2_get_boot_identity",
        "v2_get_transport_certificate",
        "v2_register_peer",
        "v2_start_tls_service",
        "v2_peer_status",
        "v2_call_peer_health",
    }
)
COORDINATOR_METHODS = frozenset(
    {
        "append_event",
        "initialize_event_signer",
        "sign_transparency_event",
        "get_event_signing_identity",
        "get_buffer",
        "clear_buffer",
        "get_buffer_size",
        "get_buffer_stats",
        "build_checkpoint",
        "sign_checkpoint",
        "receipt_verify_graph",
        "v2_provider_broker_health",
        "v2_provider_semantics_health",
        "v2_get_kms_recipient",
        "v2_get_source_add_ingress_recipient",
        "v2_seal_source_add_ingress_credential",
        "v2_get_openrouter_ingress_recipient",
        "v2_seal_openrouter_ingress_credential",
        "v2_provision_encrypted_secret",
        "v2_get_job_kms_recipient",
        "v2_provision_job_encrypted_secret",
        "v2_provision_job_sealed_source_add_secret",
        "v2_provision_job_sealed_openrouter_secret",
        "v2_release_job_credentials",
        "v2_list_encrypted_artifacts",
        "v2_export_encrypted_artifact",
        "v2_verify_encrypted_artifact_persistence",
    }
)


class RPCAuthorityError(ValueError):
    """An unknown role or cross-role RPC method was requested."""


def active_enclave_role() -> str:
    role = str(os.getenv("LEADPOET_ENCLAVE_ROLE", "") or "").strip()
    if role not in {COORDINATOR_ROLE, AUTORESEARCH_ROLE} | SCORING_ROLES:
        raise RPCAuthorityError("gateway enclave role is missing or unknown")
    return role


def allowed_exact_methods(role: str) -> FrozenSet[str]:
    if role == COORDINATOR_ROLE:
        return COMMON_METHODS | COORDINATOR_METHODS
    if role in SCORING_ROLES or role == AUTORESEARCH_ROLE:
        return COMMON_METHODS
    raise RPCAuthorityError("unknown gateway enclave role")


def rpc_method_allowed(role: str, method: str) -> bool:
    normalized_method = str(method or "")
    if normalized_method in allowed_exact_methods(role):
        return True
    if role in SCORING_ROLES:
        return normalized_method.startswith("scoring_v2_")
    if role == AUTORESEARCH_ROLE:
        return normalized_method.startswith("autoresearch_v2_")
    if role == COORDINATOR_ROLE:
        return normalized_method.startswith("coordinator_v2_")
    return False
