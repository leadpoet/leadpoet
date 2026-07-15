import pytest

from gateway.tee.rpc_authority import (
    AUTORESEARCH_ROLE,
    COORDINATOR_ROLE,
    RPCAuthorityError,
    active_enclave_role,
    rpc_method_allowed,
)


@pytest.mark.parametrize("role", ["gateway_scoring"])
def test_scoring_roles_cannot_use_coordinator_or_autoresearch_authority(role):
    assert rpc_method_allowed(role, "scoring_v2_submit_job")
    assert rpc_method_allowed(role, "role_health")
    assert not rpc_method_allowed(role, "scoring_submit_job")
    assert not rpc_method_allowed(role, "append_event")
    assert not rpc_method_allowed(role, "sign_checkpoint")
    assert not rpc_method_allowed(role, "autoresearch_submit_job")


def test_autoresearch_role_cannot_use_scoring_or_coordinator_authority():
    assert rpc_method_allowed(AUTORESEARCH_ROLE, "autoresearch_v2_submit_job")
    assert not rpc_method_allowed(AUTORESEARCH_ROLE, "autoresearch_submit_job")
    assert not rpc_method_allowed(AUTORESEARCH_ROLE, "scoring_submit_job")
    assert not rpc_method_allowed(AUTORESEARCH_ROLE, "sign_checkpoint")


def test_coordinator_is_the_only_v2_event_and_provider_authority():
    assert rpc_method_allowed(COORDINATOR_ROLE, "append_event")
    assert rpc_method_allowed(COORDINATOR_ROLE, "initialize_event_signer")
    assert rpc_method_allowed(COORDINATOR_ROLE, "sign_transparency_event")
    assert rpc_method_allowed(COORDINATOR_ROLE, "get_event_signing_identity")
    assert rpc_method_allowed(COORDINATOR_ROLE, "coordinator_v2_submit_job")
    assert not rpc_method_allowed(COORDINATOR_ROLE, "provider_execute")
    assert rpc_method_allowed(COORDINATOR_ROLE, "v2_get_kms_recipient")
    assert rpc_method_allowed(COORDINATOR_ROLE, "v2_provider_broker_health")
    assert rpc_method_allowed(COORDINATOR_ROLE, "v2_provider_semantics_health")
    assert rpc_method_allowed(
        COORDINATOR_ROLE, "v2_provision_encrypted_secret"
    )
    assert not rpc_method_allowed(
        COORDINATOR_ROLE, "v2_provision_provider_credentials"
    )
    assert rpc_method_allowed(COORDINATOR_ROLE, "receipt_verify_graph")
    assert not rpc_method_allowed(COORDINATOR_ROLE, "scoring_submit_job")
    assert not rpc_method_allowed(COORDINATOR_ROLE, "autoresearch_submit_job")
    assert not rpc_method_allowed(COORDINATOR_ROLE, "set_pcr_measurements")


@pytest.mark.parametrize(
    "role", ["gateway_scoring", AUTORESEARCH_ROLE]
)
def test_non_coordinator_roles_cannot_sign_transparency_events(role):
    assert not rpc_method_allowed(role, "initialize_event_signer")
    assert not rpc_method_allowed(role, "sign_transparency_event")
    assert not rpc_method_allowed(role, "get_event_signing_identity")


def test_missing_role_fails_closed(monkeypatch):
    monkeypatch.delenv("LEADPOET_ENCLAVE_ROLE", raising=False)
    with pytest.raises(RPCAuthorityError, match="missing or unknown"):
        active_enclave_role()


def test_unknown_role_fails_closed():
    with pytest.raises(RPCAuthorityError, match="unknown"):
        rpc_method_allowed("attacker", "get_public_key")
