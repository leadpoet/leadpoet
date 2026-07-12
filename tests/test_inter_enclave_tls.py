from __future__ import annotations

import base64
import socket
import threading

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.inter_enclave_tls import (
    AttestedPeerRegistry,
    AttestedTLSRPCClient,
    AttestedTLSRPCServer,
    InterEnclaveTLSError,
    _read_frame,
    _send_frame,
    build_rpc_request,
    validate_rpc_request,
)
from gateway.tee.mtls_identity import (
    MutualAttestationError,
    generate_ephemeral_tls_identity,
)
from leadpoet_canonical.attested_v2 import (
    build_boot_identity_body,
    create_boot_identity,
)


HASH = "sha256:" + "a" * 64


def _boot(physical_role, service_role, tls_identity, marker):
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    return create_boot_identity(
        body=build_boot_identity_body(
            role=service_role,
            physical_role=physical_role,
            commit_sha=marker * 40,
            pcr0=marker * 96,
            build_manifest_hash="sha256:" + marker * 64,
            dependency_lock_hash=HASH,
            config_hash=HASH,
            boot_nonce=marker * 32,
            signing_pubkey=pubkey,
            transport_pubkey=tls_identity["transport_pubkey"],
            transport_certificate_hash=tls_identity["certificate_sha256"],
            attestation_user_data_hash=HASH,
            issued_at="2026-07-10T20:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"nitro").decode("ascii"),
    )


def _registry(local_role):
    observed = []

    def _verify(identity, *, expected_pcr0):
        observed.append((identity["physical_role"], expected_pcr0))
        if identity["pcr0"] != expected_pcr0:
            raise ValueError("PCR mismatch")
        return {"verified": True}

    return AttestedPeerRegistry(
        local_physical_role=local_role,
        boot_verifier=_verify,
    ), observed


def _register(registry, boot, tls_identity):
    return registry.register(
        boot_identity=boot,
        certificate_pem=tls_identity["certificate_pem"],
        expected_pcr0=boot["pcr0"],
        expected_commit_sha=boot["commit_sha"],
        expected_build_manifest_hash=boot["build_manifest_hash"],
        expected_config_hash=boot["config_hash"],
    )


def test_peer_registry_requires_exact_nitro_release_and_certificate():
    coordinator_tls = generate_ephemeral_tls_identity(
        service_role="gateway_coordinator"
    )
    coordinator_boot = _boot(
        "gateway_coordinator", "gateway_coordinator", coordinator_tls, "b"
    )
    registry, observed = _registry("gateway_scoring_a")
    registered = _register(registry, coordinator_boot, coordinator_tls)
    assert registered["physical_role"] == "gateway_coordinator"
    assert observed == [("gateway_coordinator", "b" * 96)]

    with pytest.raises(InterEnclaveTLSError, match="commit"):
        registry.register(
            boot_identity=coordinator_boot,
            certificate_pem=coordinator_tls["certificate_pem"],
            expected_pcr0=coordinator_boot["pcr0"],
            expected_commit_sha="c" * 40,
            expected_build_manifest_hash=coordinator_boot["build_manifest_hash"],
        )

    substituted = generate_ephemeral_tls_identity(service_role="gateway_coordinator")
    with pytest.raises(MutualAttestationError, match="attested"):
        registry.register(
            boot_identity=coordinator_boot,
            certificate_pem=substituted["certificate_pem"],
            expected_pcr0=coordinator_boot["pcr0"],
            expected_commit_sha=coordinator_boot["commit_sha"],
            expected_build_manifest_hash=coordinator_boot["build_manifest_hash"],
        )


def test_registry_rejects_runner_to_runner_channel():
    scoring_tls = generate_ephemeral_tls_identity(service_role="gateway_scoring")
    scoring_boot = _boot(
        "gateway_scoring_b", "gateway_scoring", scoring_tls, "c"
    )
    registry, _ = _registry("gateway_scoring_a")
    with pytest.raises(InterEnclaveTLSError, match="pair"):
        _register(registry, scoring_boot, scoring_tls)


def test_rpc_request_binds_both_boots_and_topology():
    request = build_rpc_request(
        method="provider_execute",
        params={"job_id": "job-1"},
        channel_id="d" * 32,
        source_boot_identity_hash=HASH,
        target_boot_identity_hash="sha256:" + "e" * 64,
    )
    assert validate_rpc_request(
        request,
        source_boot_identity_hash=HASH,
        target_boot_identity_hash="sha256:" + "e" * 64,
    ) == request
    changed = dict(request)
    changed["target_boot_identity_hash"] = "sha256:" + "f" * 64
    with pytest.raises(InterEnclaveTLSError, match="target boot"):
        validate_rpc_request(
            changed,
            source_boot_identity_hash=HASH,
            target_boot_identity_hash="sha256:" + "e" * 64,
        )


def test_tls_rpc_round_trip_uses_only_attested_peer_certificates(tmp_path):
    import ssl

    if not getattr(ssl, "HAS_TLSv1_3", False):
        pytest.skip("TLS 1.3 unavailable")

    coordinator_tls = generate_ephemeral_tls_identity(
        service_role="gateway_coordinator"
    )
    scoring_tls = generate_ephemeral_tls_identity(service_role="gateway_scoring")
    coordinator_boot = _boot(
        "gateway_coordinator", "gateway_coordinator", coordinator_tls, "b"
    )
    scoring_boot = _boot(
        "gateway_scoring_a", "gateway_scoring", scoring_tls, "c"
    )
    coordinator_registry, _ = _registry("gateway_coordinator")
    scoring_registry, _ = _registry("gateway_scoring_a")
    _register(coordinator_registry, scoring_boot, scoring_tls)
    _register(scoring_registry, coordinator_boot, coordinator_tls)

    server = AttestedTLSRPCServer(
        local_physical_role="gateway_coordinator",
        local_boot_identity=coordinator_boot,
        local_tls_identity=coordinator_tls,
        peer_registry=coordinator_registry,
        handler=lambda method, params, peer: {
            "method": method,
            "job_id": params["job_id"],
            "peer_role": peer["physical_role"],
        },
        tmpfs_root=tmp_path / "server",
    )

    errors = []

    def _connector():
        client_socket, relay_socket = socket.socketpair()

        def _relay_target():
            try:
                control = _read_frame(relay_socket)
                _send_frame(
                    relay_socket,
                    {
                        "result": {
                            "status": "connected",
                            "channel_id": control["channel_id"],
                        }
                    },
                )
                server.handle_connection(relay_socket)
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        threading.Thread(target=_relay_target, daemon=True).start()
        return client_socket

    client = AttestedTLSRPCClient(
        local_physical_role="gateway_scoring_a",
        local_boot_identity=scoring_boot,
        local_tls_identity=scoring_tls,
        peer_registry=scoring_registry,
        tmpfs_root=tmp_path / "client",
        connector=_connector,
    )
    try:
        result = client.call(
            target_physical_role="gateway_coordinator",
            method="provider_execute",
            params={"job_id": "job-1"},
            channel_id="f" * 32,
        )
    except Exception as exc:
        if "TLS 1.3 is unavailable" in str(exc):
            pytest.skip(str(exc))
        raise
    assert not errors
    assert result == {
        "method": "provider_execute",
        "job_id": "job-1",
        "peer_role": "gateway_scoring_a",
    }
