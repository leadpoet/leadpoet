"""A public proof refresh must not alter the enclave's pinned boot identity."""

from __future__ import annotations

import base64

import pytest

from gateway.api import attestation
from gateway.tee import tee_service


def test_existing_identity_rpc_refreshes_only_on_exact_opt_in(monkeypatch):
    calls = []

    class Manager:
        def boot_identity(self):
            return {
                "signing_pubkey": "a" * 64,
                "physical_role": "gateway_coordinator",
                "commit_sha": "b" * 40,
                "boot_identity_hash": "sha256:" + "c" * 64,
                "attestation_document_b64": base64.b64encode(b"boot").decode(),
            }

        def fresh_attestation_document(self):
            calls.append("refresh")
            return b"fresh"

    monkeypatch.setenv("LEADPOET_ENCLAVE_ROLE", "gateway_coordinator")
    monkeypatch.setattr(tee_service, "get_v2_runtime_identity", lambda: Manager())
    monkeypatch.setattr(tee_service, "compute_code_hash", lambda: "d" * 64)

    default = tee_service.handle_rpc("get_event_signing_identity", {})["result"]
    assert default["attestation_document_b64"] == base64.b64encode(b"boot").decode()
    assert calls == []
    fresh = tee_service.handle_rpc("get_event_signing_identity", {"fresh_attestation": True})["result"]
    assert fresh["attestation_document_b64"] == base64.b64encode(b"fresh").decode()
    assert fresh["enclave_pubkey"] == default["enclave_pubkey"]
    assert fresh["signer_state"] == default["signer_state"]
    assert calls == ["refresh"]
    for invalid in (False, 1, "true", None):
        assert tee_service.handle_rpc(
            "get_event_signing_identity", {"fresh_attestation": invalid}
        )["error_type"] == "ValueError"
    assert tee_service.handle_rpc(
        "get_event_signing_identity", {"fresh_attestation": True, "extra": True}
    )["error_type"] == "ValueError"


@pytest.mark.asyncio
async def test_public_document_opts_in_while_default_identity_reader_does_not(monkeypatch):
    calls = []

    class Client:
        async def get_event_signing_identity(self, *, fresh_attestation=False):
            calls.append(fresh_attestation)
            return {
                "purpose": "gateway_event_signing",
                "enclave_pubkey": "a" * 64,
                "code_hash": "b" * 64,
                "attestation_document_b64": "fresh" if fresh_attestation else "boot",
            }

    monkeypatch.setattr(attestation, "coordinator_tee_client", Client())
    assert (await attestation._runtime_identity())["attestation_document_b64"] == "boot"
    response = await attestation.get_attestation_document_endpoint()
    assert response.attestation_document == "fresh"
    assert calls == [False, True]
