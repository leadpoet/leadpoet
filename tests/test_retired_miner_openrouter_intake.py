from gateway.tee.coordinator_executor_v2 import COORDINATOR_OPERATIONS_V2
from gateway.tee.provider_broker_v2 import (
    BUILTIN_PROVIDER_ROUTES,
    expected_job_credential_slot_ref_hashes,
    expected_provider_credential_slots,
)
from gateway.tee.rpc_authority import COORDINATOR_ROLE, rpc_method_allowed
from gateway.utils.tee_client import TEEClient


def test_miner_openrouter_credential_intake_is_not_exposed() -> None:
    assert not rpc_method_allowed(
        COORDINATOR_ROLE, "v2_get_openrouter_ingress_recipient"
    )
    assert not rpc_method_allowed(
        COORDINATOR_ROLE, "v2_seal_openrouter_ingress_credential"
    )
    assert not rpc_method_allowed(
        COORDINATOR_ROLE, "v2_provision_job_sealed_openrouter_secret"
    )
    assert not hasattr(TEEClient, "v2_get_openrouter_ingress_recipient")
    assert not hasattr(TEEClient, "v2_seal_openrouter_ingress_credential")
    assert not hasattr(
        TEEClient, "v2_provision_job_sealed_openrouter_secret"
    )
    assert "register_openrouter_credential_v2" not in COORDINATOR_OPERATIONS_V2
    assert "preflight_openrouter_credential_v2" not in COORDINATOR_OPERATIONS_V2
    assert "openrouter_management" not in BUILTIN_PROVIDER_ROUTES
    assert "openrouter_management" not in expected_job_credential_slot_ref_hashes()


def test_source_ingress_is_removed_but_shared_provider_provisioning_remains() -> None:
    assert not rpc_method_allowed(
        COORDINATOR_ROLE, "v2_get_source_add_ingress_recipient"
    )
    assert not rpc_method_allowed(
        COORDINATOR_ROLE, "v2_seal_source_add_ingress_credential"
    )
    assert rpc_method_allowed(
        COORDINATOR_ROLE, "v2_provision_encrypted_secret"
    )
    assert not rpc_method_allowed(
        COORDINATOR_ROLE, "v2_provision_job_sealed_source_add_secret"
    )
    assert not hasattr(TEEClient, "v2_get_source_add_ingress_recipient")
    assert not hasattr(TEEClient, "v2_seal_source_add_ingress_credential")
    assert hasattr(TEEClient, "v2_provision_encrypted_secret")
    assert not hasattr(TEEClient, "v2_provision_job_sealed_source_add_secret")
    assert "openrouter" in expected_provider_credential_slots()
