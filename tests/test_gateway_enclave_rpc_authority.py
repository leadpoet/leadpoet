import pytest

from gateway.tee.rpc_authority import (
    COORDINATOR_ROLE,
    RPCAuthorityError,
    active_enclave_role,
    allowed_exact_methods,
    rpc_method_allowed,
)


def test_coordinator_exposes_only_measured_identity_rpc_surface():
    assert allowed_exact_methods(COORDINATOR_ROLE) == {
        "role_health",
        "v2_configure_runtime",
        "v2_get_boot_identity",
        "get_event_signing_identity",
    }
    assert rpc_method_allowed(COORDINATOR_ROLE, "get_event_signing_identity")
    assert not rpc_method_allowed(COORDINATOR_ROLE, "provider_execute")
    assert not rpc_method_allowed(COORDINATOR_ROLE, "scoring_v2_submit_job")
    assert not rpc_method_allowed(COORDINATOR_ROLE, "v2_register_peer")


def test_missing_or_retired_role_fails_closed(monkeypatch):
    monkeypatch.delenv("LEADPOET_ENCLAVE_ROLE", raising=False)
    with pytest.raises(RPCAuthorityError, match="missing or unknown"):
        active_enclave_role()
    monkeypatch.setenv("LEADPOET_ENCLAVE_ROLE", "gateway_scoring")
    with pytest.raises(RPCAuthorityError, match="missing or unknown"):
        active_enclave_role()
    with pytest.raises(RPCAuthorityError, match="unknown"):
        rpc_method_allowed("gateway_scoring", "role_health")
